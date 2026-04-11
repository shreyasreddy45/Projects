import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

import './bloc/personal_details_bloc.dart';
import './bloc/personal_details_event.dart';
import './bloc/personal_details_state.dart';
import './models/personal_details_model.dart';

class PersonalDetailsPage extends StatefulWidget {
  final String userId;

  const PersonalDetailsPage({super.key, required this.userId});

  @override
  State<PersonalDetailsPage> createState() => _PersonalDetailsPageState();
}

class _PersonalDetailsPageState extends State<PersonalDetailsPage> {
  final _formKey = GlobalKey<FormState>();
  bool sameAsPresent = false;

  final dobController = TextEditingController();
  final genderController = TextEditingController();
  final bloodGroupController = TextEditingController();
  final altPhoneController = TextEditingController();
  final emailController = TextEditingController();
  final aadharController = TextEditingController();

  final presentFlatController = TextEditingController();
  final presentStreetController = TextEditingController();
  final presentCityController = TextEditingController();
  final presentStateController = TextEditingController();
  final presentPincodeController = TextEditingController();

  final permanentFlatController = TextEditingController();
  final permanentStreetController = TextEditingController();
  final permanentCityController = TextEditingController();
  final permanentStateController = TextEditingController();
  final permanentPincodeController = TextEditingController();

  @override
  void dispose() {
    dobController.dispose();
    genderController.dispose();
    bloodGroupController.dispose();
    altPhoneController.dispose();
    emailController.dispose();
    aadharController.dispose();
    presentFlatController.dispose();
    presentStreetController.dispose();
    presentCityController.dispose();
    presentStateController.dispose();
    presentPincodeController.dispose();
    permanentFlatController.dispose();
    permanentStreetController.dispose();
    permanentCityController.dispose();
    permanentStateController.dispose();
    permanentPincodeController.dispose();
    super.dispose();
  }

  void _copyAddress() {
    if (sameAsPresent) {
      permanentFlatController.text = presentFlatController.text;
      permanentStreetController.text = presentStreetController.text;
      permanentCityController.text = presentCityController.text;
      permanentStateController.text = presentStateController.text;
      permanentPincodeController.text = presentPincodeController.text;
    } else {
      permanentFlatController.clear();
      permanentStreetController.clear();
      permanentCityController.clear();
      permanentStateController.clear();
      permanentPincodeController.clear();
    }
  }

  @override
  Widget build(BuildContext context) {
    return BlocProvider(
      create: (_) => PersonalDetailsBloc(),
      child: Scaffold(
        appBar: AppBar(title: const Text('Personal Details')),
        body: BlocListener<PersonalDetailsBloc, PersonalDetailsState>(
          listener: (context, state) {
            if (state is PersonalDetailsSuccess) {
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(
                    content: Text("Personal details updated successfully!")),
              );
              // Optional: Navigate to next page or pop
            } else if (state is PersonalDetailsFailure) {
              ScaffoldMessenger.of(context).showSnackBar(
                SnackBar(content: Text("Failed to update: ${state.error}")),
              );
            }
          },
          child: SafeArea(
            child: SingleChildScrollView(
              padding: const EdgeInsets.all(16),
              child: Form(
                key: _formKey,
                child: Column(
                  children: [
                    _buildField('Date of Birth', dobController),
                    _buildField('Gender', genderController),
                    _buildField('Blood Group', bloodGroupController),
                    _buildField('Alternate Phone', altPhoneController,
                        hint: '10-digit number'),
                    _buildField('Personal Email', emailController,
                        hint: 'example@gmail.com'),
                    _buildField('Aadhar Number', aadharController,
                        hint: '12-digit number'),
                    const SizedBox(height: 16),
                    const Align(
                      alignment: Alignment.centerLeft,
                      child: Text(
                        "PRESENT ADDRESS",
                        style: TextStyle(
                            color: Colors.blue, fontWeight: FontWeight.bold),
                      ),
                    ),
                    _buildField('Flat/Apartment', presentFlatController),
                    _buildField('Street', presentStreetController),
                    _buildField('City', presentCityController),
                    _buildField('State', presentStateController),
                    _buildField('Pincode', presentPincodeController),
                    const SizedBox(height: 16),
                    Row(
                      children: [
                        const Text(
                          "PERMANENT ADDRESS",
                          style: TextStyle(
                              color: Colors.blue, fontWeight: FontWeight.bold),
                        ),
                        const Spacer(),
                        Checkbox(
                          value: sameAsPresent,
                          onChanged: (value) {
                            setState(() {
                              sameAsPresent = value ?? false;
                              _copyAddress();
                            });
                          },
                        ),
                        const Text("Same as Present"),
                      ],
                    ),
                    _buildField('Flat/Apartment', permanentFlatController),
                    _buildField('Street', permanentStreetController),
                    _buildField('City', permanentCityController),
                    _buildField('State', permanentStateController),
                    _buildField('Pincode', permanentPincodeController),
                    const SizedBox(height: 24),
                    BlocBuilder<PersonalDetailsBloc, PersonalDetailsState>(
                      builder: (context, state) => ElevatedButton(
                        onPressed: state is PersonalDetailsSubmitting
                            ? null
                            : () {
                                if (_formKey.currentState?.validate() ??
                                    false) {
                                  final model = PersonalDetailsModel(
                                    dob: dobController.text.trim(),
                                    gender: genderController.text.trim(),
                                    bloodGroup:
                                        bloodGroupController.text.trim(),
                                    altPhone: altPhoneController.text.trim(),
                                    personalEmail: emailController.text.trim(),
                                    aadhar: aadharController.text.trim(),
                                    presentAddress: {
                                      'flat': presentFlatController.text.trim(),
                                      'street':
                                          presentStreetController.text.trim(),
                                      'city': presentCityController.text.trim(),
                                      'state':
                                          presentStateController.text.trim(),
                                      'pincode':
                                          presentPincodeController.text.trim(),
                                    },
                                    permanentAddress: {
                                      'flat':
                                          permanentFlatController.text.trim(),
                                      'street':
                                          permanentStreetController.text.trim(),
                                      'city':
                                          permanentCityController.text.trim(),
                                      'state':
                                          permanentStateController.text.trim(),
                                      'pincode': permanentPincodeController.text
                                          .trim(),
                                    },
                                  );
                                  context.read<PersonalDetailsBloc>().add(
                                        SubmitPersonalDetails(
                                            model, widget.userId),
                                      );
                                }
                              },
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.red,
                        ),
                        child: state is PersonalDetailsSubmitting
                            ? const SizedBox(
                                height: 16,
                                width: 16,
                                child: CircularProgressIndicator(
                                    strokeWidth: 2, color: Colors.white),
                              )
                            : const Text("Update"),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildField(String label, TextEditingController controller,
          {String? hint}) =>
      Padding(
        padding: const EdgeInsets.symmetric(vertical: 8),
        child: TextFormField(
          controller: controller,
          decoration: InputDecoration(
            labelText: '$label *',
            hintText: hint,
            border: const OutlineInputBorder(),
          ),
          validator: (value) =>
              value == null || value.trim().isEmpty ? 'Required' : null,
        ),
      );
}
